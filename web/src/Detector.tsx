import React from 'react'
import { Grid, Typography, makeStyles, Paper } from "@material-ui/core"
import CodeForm from './CodeForm';

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

export interface ICodeVulnerableProps {
    isVulnerable?: boolean;
    vulnerablityCategory?: string;
}

export interface IDetectorProps extends ICodeVulnerableProps {
    backendAlive: boolean;
}

const Detector = (props: IDetectorProps) => {
    const classes = useStyles();

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
                <Grid item xs={12}>
                    <Paper className={`${classes.section} ${classes.space} ${classes.paper}`}>
                        <CodeForm />
                    </Paper>
                </Grid>
            </Grid>
        </div>
    )
}

export default Detector