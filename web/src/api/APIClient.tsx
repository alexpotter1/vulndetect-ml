import axios, { Method } from 'axios';

const BASE_URI = 'http://localhost:5000'

const client = axios.create({
    baseURL: BASE_URI
});

export interface IAPIResponse {
    status: string,
    message: string,
}

class APIClient {
    async do(method: Method, resource: (string | undefined), data?: any): Promise<IAPIResponse> {
        return client({
            method,
            url: resource,
            data: data
        }).then(res => {
            return res.data as IAPIResponse
        })
    }

    getFlaskHello() {
        return this.do('get', '/api')
    }
}

export default APIClient;