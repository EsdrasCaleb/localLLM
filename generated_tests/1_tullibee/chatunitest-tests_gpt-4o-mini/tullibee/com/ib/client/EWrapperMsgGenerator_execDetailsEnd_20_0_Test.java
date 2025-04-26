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
        // Arrange
        EWrapperMsgGenerator generator = new EWrapperMsgGenerator();
        int reqId = 123;
        // Act
        String result = (String) EWrapperMsgGenerator.class.getDeclaredMethod("execDetailsEnd", int.class).invoke(generator, reqId);
        // Assert
        assertEquals("reqId = [123] =============== end ===============", result);
    }
}
