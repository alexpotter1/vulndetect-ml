import React, { useState } from 'react'
import { Grid, Typography, makeStyles, Paper, CircularProgress } from "@material-ui/core"
import CodeForm from './CodeForm';
import { IAPIResponse } from './api/APIClient';

const useStyles = makeStyles((theme) => ({
    root: {
        flexGrow: 1,
    },
    section: {
        textAlign: 'center',
    },
    space: {
        marginLeft: theme.spacing(5),
        marginRight: theme.spacing(5),
        marginTop: theme.spacing(3),
        marginBottom: theme.spacing(3),
        padding: theme.spacing(3),
    },
    paper: {
        color: theme.palette.text.secondary,
    },
}));

interface IFormErrors {
    code: string;
}

export interface ICodeVulnerable {
    isVulnerable?: boolean;
    vulnerablityCategory?: string;
    predictionConfidence?: number;
}

export interface IDetectorProps {
    backendAlive: boolean;
}

const Detector = (props: IDetectorProps) => {
    const classes = useStyles();
    const [analysisResponse, setAnalysisResponse] = useState<IAPIResponse>({});
    const notifyCallback = (res: IAPIResponse) => { setAnalysisResponse(res) }

    return (
        <div className={classes.root}>
            <Grid container spacing={3}>
                <Grid item xs={12}>
                    <Typography className={`${classes.section} ${classes.space}`} variant="h3">
                        Find out if your Java code is vulnerable!
                    </Typography>
                    <Typography className={classes.section} variant="h5">
                        By Alex Potter (6416626)
                    </Typography>
                </Grid>
                <Grid item xs={12}>
                    <span className={classes.section}>
                    <Typography variant="h6">
                        Backend status: 
                    </Typography>
                    <Typography variant="h6" color={props.backendAlive ? 'primary' : 'error'}>
                        {props.backendAlive ? 'Connected' : 'Not connected'}
                    </Typography>
                    </span>
                </Grid>
                {analysisResponse.status && 
                <Grid item xs={12}>
                    <Paper className={`${classes.section} ${classes.space} ${classes.paper}`}>
                        <Typography variant="h5">Code Analysis</Typography>
                        {analysisResponse.status === 'waiting' &&
                        <>
                            <Typography variant="h6">Waiting for response...</Typography>
                            <CircularProgress color="secondary" />
                        </>}
                        {analysisResponse.status !== '200 OK' && analysisResponse.status !== 'waiting' &&
                        <>
                            <Typography variant="h6" color={'secondary'}>{analysisResponse.status}</Typography>
                            <Typography variant="h6">Reason: {analysisResponse.message}</Typography>
                        </>}
                        {analysisResponse.status === '200 OK' &&
                        <Typography variant="h6" color={analysisResponse.isVulnerable ? 'secondary' : 'primary'}>{analysisResponse.isVulnerable ? 'Vulnerable' : 'Not vulnerable'}</Typography>}
                        {analysisResponse.vulnerablityCategory !== undefined && <Typography variant="h6">Category: {analysisResponse.vulnerablityCategory}</Typography>}
                        {analysisResponse.predictionConfidence !== undefined && <Typography variant="h6">Confidence: {Math.round(analysisResponse.predictionConfidence)}%</Typography>}
                    </Paper>
                </Grid>}
                <Grid item xs={12}>
                    <Paper className={`${classes.section} ${classes.space} ${classes.paper}`}>
                        <CodeForm notify={notifyCallback} />
                    </Paper>
                </Grid>
            </Grid>
        </div>
    )
}

export default Detector