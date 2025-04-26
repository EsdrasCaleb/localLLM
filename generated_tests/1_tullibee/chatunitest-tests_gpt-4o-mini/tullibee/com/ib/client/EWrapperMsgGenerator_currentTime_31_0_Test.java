package com.ib.client;

import java.lang.reflect.Method;
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
        // Arrange
        // Represents 2021-10-01 00:00:00 UTC
        long inputTime = 1633036800L;
        String expectedOutput = "current time = " + inputTime + " (" + DateFormat.getDateTimeInstance().format(new Date(inputTime * 1000)) + ")";
        // Act
        Method method = EWrapperMsgGenerator.class.getDeclaredMethod("currentTime", long.class);
        method.setAccessible(true);
        String actualOutput = (String) method.invoke(null, inputTime);
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }
}
