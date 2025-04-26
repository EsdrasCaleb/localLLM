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

public class EWrapperMsgGenerator_currentTime_31_1_Test {

    @Test
    public void testCurrentTime() {
        long testTime = 1614096000L;
        String expectedResult = "current time = 1614096000 (Wednesday, June 21, 2021, 12:00:00 AM)";
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        String result = eWrapperMsgGenerator.currentTime(testTime);
        assertEquals(expectedResult, result);
    }
}
