package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_scannerDataEnd_30_0_Test {

    @InjectMocks
    private EWrapperMsgGenerator eWrapperMsgGenerator;

    @Test
    public void testScannerDataEnd() throws Exception {
        int reqId = 123;
        String expected = "id = 123 =============== end ===============";
        // Use reflection to invoke the private method
        java.lang.reflect.Method method = EWrapperMsgGenerator.class.getDeclaredMethod("scannerDataEnd", int.class);
        method.setAccessible(true);
        String result = (String) method.invoke(eWrapperMsgGenerator, reqId);
        assertEquals(expected, result);
    }
}
