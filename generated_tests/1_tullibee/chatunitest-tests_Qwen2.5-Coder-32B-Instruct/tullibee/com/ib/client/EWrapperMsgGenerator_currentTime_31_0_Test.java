package com.ib.client;

import java.lang.reflect.Method;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
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
        // Test data
        // Corresponds to 2021-10-01 00:00:00 GMT
        long testTimeInSeconds = 1633072800;
        String expectedFormattedDate = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date(testTimeInSeconds * 1000));
        // Get the method using reflection
        Method method = EWrapperMsgGenerator.class.getDeclaredMethod("currentTime", long.class);
        method.setAccessible(true);
        // Invoke the method
        String result = (String) method.invoke(null, testTimeInSeconds);
        // Expected result
        String expectedResult = "current time = " + testTimeInSeconds + " (" + expectedFormattedDate + ")";
        // Assert the result
        assertEquals(expectedResult, result);
    }
}
