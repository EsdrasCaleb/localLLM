package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class AnyWrapperMsgGenerator_error_2_0_Test {

    @Test
    public void testErrorMethod() {
        int id = 123;
        int errorCode = 404;
        String errorMsg = "Not Found";
        String expected = Integer.toString(id) + " | " + Integer.toString(errorCode) + " | " + errorMsg;
        String result = AnyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        assertEquals(expected, result);
    }
}
