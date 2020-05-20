import axios, { Method } from 'axios';
import { ICodeVulnerableProps } from '../Detector';

const BASE_URI = 'http://localhost:5000'

const client = axios.create({
    baseURL: BASE_URI
});

export interface IAPIResponse extends ICodeVulnerableProps {
    status: string,
    message: string,
}

var APIClient = {
    do: async (method: Method, resource: (string | undefined), data?: any): Promise<IAPIResponse> => {
        return client({
            method,
            url: resource,
            data: data
        }).then(res => {
            return res.data as IAPIResponse
        })
    },

    getBackendHeartbeat: (): Promise<IAPIResponse> => {
        return APIClient.do('get', '/api/heartbeat')
    },

    postCodeSample: (data: FormData): Promise<IAPIResponse> => {
        return APIClient.do('post', '/api/predict', data)
    }
}

export default APIClient;