package com.ib.client;

import java.lang.reflect.Field;
import java.text.DateFormat;
import java.util.Date;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;

public class EWrapperMsgGenerator_currentTime_31_0_Test {

    @Test
    public void testCurrentTime() throws Exception {
        // Example Unix timestamp
        long time = 1633072800;
        String expected = "current time = " + time + " (" + DateFormat.getDateTimeInstance().format(new Date(time * 1000)) + ")";
        String result = EWrapperMsgGenerator.currentTime(time);
        assertEquals(expected, result);
    }

    @Test
    public void testCurrentTimeWithNegativeValue() throws Exception {
        // Example negative Unix timestamp
        long time = -1633072800;
        String expected = "current time = " + time + " (" + DateFormat.getDateTimeInstance().format(new Date(time * 1000)) + ")";
        String result = EWrapperMsgGenerator.currentTime(time);
        assertEquals(expected, result);
    }

    @Test
    public void testCurrentTimeWithZero() throws Exception {
        // Unix timestamp for epoch
        long time = 0;
        String expected = "current time = " + time + " (" + DateFormat.getDateTimeInstance().format(new Date(time * 1000)) + ")";
        String result = EWrapperMsgGenerator.currentTime(time);
        assertEquals(expected, result);
    }

    @Test
    public void testCurrentTimeWithMaxLong() throws Exception {
        // Maximum possible Unix timestamp
        long time = Long.MAX_VALUE;
        String expected = "current time = " + time + " (" + DateFormat.getDateTimeInstance().format(new Date(time * 1000)) + ")";
        String result = EWrapperMsgGenerator.currentTime(time);
        assertEquals(expected, result);
    }

    @Test
    public void testCurrentTimeWithMinLong() throws Exception {
        // Minimum possible Unix timestamp
        long time = Long.MIN_VALUE;
        String expected = "current time = " + time + " (" + DateFormat.getDateTimeInstance().format(new Date(time * 1000)) + ")";
        String result = EWrapperMsgGenerator.currentTime(time);
        assertEquals(expected, result);
    }
}
