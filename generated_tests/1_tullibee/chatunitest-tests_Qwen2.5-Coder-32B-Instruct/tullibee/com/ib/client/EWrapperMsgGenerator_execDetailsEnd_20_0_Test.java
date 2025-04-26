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

public class EWrapperMsgGenerator_execDetailsEnd_20_0_Test {

    @Test
    public void testExecDetailsEnd() throws Exception {
        // Create a mock instance of EWrapperMsgGenerator
        EWrapperMsgGenerator eWrapperMsgGenerator = Mockito.mock(EWrapperMsgGenerator.class);
        // Use reflection to call the private method execDetailsEnd
        java.lang.reflect.Method method = EWrapperMsgGenerator.class.getDeclaredMethod("execDetailsEnd", int.class);
        method.setAccessible(true);
        // Test with a specific reqId
        int reqId = 12345;
        String expectedMessage = "reqId = 12345 =============== end ===============";
        String actualMessage = (String) method.invoke(eWrapperMsgGenerator, reqId);
        // Verify the result
        assertEquals(expectedMessage, actualMessage);
    }
}
