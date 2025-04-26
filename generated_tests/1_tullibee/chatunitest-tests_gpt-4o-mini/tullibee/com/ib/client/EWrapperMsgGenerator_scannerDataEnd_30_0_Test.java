package com.ib.client;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

public class EWrapperMsgGenerator_scannerDataEnd_30_0_Test {

    @Test
    public void testScannerDataEnd() throws Exception {
        // Arrange
        EWrapperMsgGenerator generator = new EWrapperMsgGenerator();
        int reqId = 123;
        // Access the private method using reflection
        Method method = EWrapperMsgGenerator.class.getDeclaredMethod("scannerDataEnd", int.class);
        method.setAccessible(true);
        // Act
        String result = (String) method.invoke(generator, reqId);
        // Assert
        String expected = "id = [123] =============== end ===============";
        assertEquals(expected, result);
    }
}
