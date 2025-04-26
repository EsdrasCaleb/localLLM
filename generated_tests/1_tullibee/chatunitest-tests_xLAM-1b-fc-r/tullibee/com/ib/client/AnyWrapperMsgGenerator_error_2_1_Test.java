package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_2_1_Test {

    @Test
    public void testError() {
        int id = 123;
        int errorCode = 404;
        String errorMsg = "Not Found";
        String expected = Integer.toString(id) + " | " + Integer.toString(errorCode) + " | " + errorMsg;
        String actual = AnyWrapperMsgGenerator.error(id, errorCode, errorMsg);
        assertEquals(expected, actual);
    }
}
