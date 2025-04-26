package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_2_0_Test {

    @Test
    void testError_normalCase() {
        String expected = "123 | 456 | This is an error message";
        String actual = AnyWrapperMsgGenerator.error(123, 456, "This is an error message");
        assertEquals(expected, actual);
    }

    @Test
    void testError_errorCodeZero() {
        String expected = "1 | 0 | Error";
        String actual = AnyWrapperMsgGenerator.error(1, 0, "Error");
        assertEquals(expected, actual);
    }

    @Test
    void testError_idZero() {
        String expected = "0 | 123 | Another error";
        String actual = AnyWrapperMsgGenerator.error(0, 123, "Another error");
        assertEquals(expected, actual);
    }

    @Test
    void testError_emptyErrorMessage() {
        String expected = "42 | 404 | ";
        String actual = AnyWrapperMsgGenerator.error(42, 404, "");
        assertEquals(expected, actual);
    }

    @Test
    void testError_nullErrorMessage() {
        String expected = "777 | 500 | null";
        String actual = AnyWrapperMsgGenerator.error(777, 500, null);
        assertEquals(expected, actual);
    }

    @Test
    void testError_negativeIdAndErrorCode() {
        String expected = "-1 | -100 | Negative error";
        String actual = AnyWrapperMsgGenerator.error(-1, -100, "Negative error");
        assertEquals(expected, actual);
    }

    @Test
    void testError_largeNumbers() {
        String expected = "2147483647 | 2147483647 | Very large error";
        String actual = AnyWrapperMsgGenerator.error(2147483647, 2147483647, "Very large error");
        assertEquals(expected, actual);
    }
}
