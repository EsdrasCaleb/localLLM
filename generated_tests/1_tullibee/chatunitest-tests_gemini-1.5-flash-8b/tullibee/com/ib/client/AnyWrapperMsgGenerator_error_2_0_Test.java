package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class AnyWrapperMsgGenerator_error_2_0_Test {

    @Test
    void testError_validInput() {
        String expected = "123 | 456 | This is an error message";
        String actual = AnyWrapperMsgGenerator.error(123, 456, "This is an error message");
        assertEquals(expected, actual);
    }

    @Test
    void testError_zeroInput() {
        String expected = "0 | 0 | This is another error";
        String actual = AnyWrapperMsgGenerator.error(0, 0, "This is another error");
        assertEquals(expected, actual);
    }

    @Test
    void testError_emptyMessage() {
        String expected = "1 | 2 | ";
        String actual = AnyWrapperMsgGenerator.error(1, 2, "");
        assertEquals(expected, actual);
    }

    @Test
    void testError_nullMessage() {
        String expected = "1 | 2 | null";
        String actual = AnyWrapperMsgGenerator.error(1, 2, null);
        assertEquals(expected, actual);
    }
}
