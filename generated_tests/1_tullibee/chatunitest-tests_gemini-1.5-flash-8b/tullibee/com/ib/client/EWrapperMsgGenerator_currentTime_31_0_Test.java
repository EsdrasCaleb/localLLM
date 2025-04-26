package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

class EWrapperMsgGenerator_currentTime_31_0_Test {

    @Test
    void testCurrentTime_validInput() {
        long timestamp = 1678886400;
        String expectedOutput = "current time = 1678886400 (Mar 15, 2023 12:00:00 AM)";
        String actualOutput = EWrapperMsgGenerator.currentTime(timestamp);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testCurrentTime_zeroInput() {
        long timestamp = 0;
        String expectedOutput = "current time = 0 (Jan 1, 1970 12:00:00 AM)";
        String actualOutput = EWrapperMsgGenerator.currentTime(timestamp);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testCurrentTime_negativeInput() {
        long timestamp = -1;
        String expectedOutput = "current time = -1 (Dec 31, 1969 11:59:59 PM)";
        String actualOutput = EWrapperMsgGenerator.currentTime(-1);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void testCurrentTime_largePositiveInput() {
        // Corrected to long literal
        long timestamp = 2500000000L;
        String expectedOutput = "current time = 2500000000 (Jan 19, 2070 04:00:00 AM)";
        String actualOutput = EWrapperMsgGenerator.currentTime(timestamp);
        assertEquals(expectedOutput, actualOutput);
    }
}
