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

public class EWrapperMsgGenerator_scannerDataEnd_30_0_Test {

    @Test
    public void testScannerDataEnd() throws Exception {
        // Arrange
        EWrapperMsgGenerator eWrapperMsgGenerator = Mockito.spy(new EWrapperMsgGenerator());
        int reqId = 12345;
        String expectedMessage = "id = 12345 =============== end ===============";
        // Act
        String result = eWrapperMsgGenerator.scannerDataEnd(reqId);
        // Assert
        assertEquals(expectedMessage, result);
    }

    // Assuming scannerDataEnd is a private method, we use reflection to invoke it
    @Test
    public void testPrivateScannerDataEnd() throws Exception {
        // Arrange
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        int reqId = 67890;
        String expectedMessage = "id = 67890 =============== end ===============";
        // Use reflection to access the private method
        java.lang.reflect.Method method = EWrapperMsgGenerator.class.getDeclaredMethod("scannerDataEnd", int.class);
        method.setAccessible(true);
        // Act
        String result = (String) method.invoke(eWrapperMsgGenerator, reqId);
        // Assert
        assertEquals(expectedMessage, result);
    }
}
