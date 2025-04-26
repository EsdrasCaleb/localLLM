package com.ib.client;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class AnyWrapperMsgGenerator_error_0_1_Test {

    @Test
    void testErrorWithNullException() {
        Exception ex = null;
        String expected = "Error - null";
        String actual = AnyWrapperMsgGenerator.error(ex);
        assertEquals(expected, actual);
    }

    @Test
    void testErrorWithSpecificException() {
        Exception ex = new IllegalArgumentException("Invalid input");
        String expected = "Error - Invalid input";
        String actual = AnyWrapperMsgGenerator.error(ex);
        assertEquals(expected, actual);
    }

    @Test
    void testErrorWithCustomException() {
        Exception ex = new CustomException("Something went wrong");
        String expected = "Error - Something went wrong";
        String actual = AnyWrapperMsgGenerator.error(ex);
        assertEquals(expected, actual);
    }

    // Example of a custom exception
    static class CustomException extends Exception {

        public CustomException(String message) {
            super(message);
        }
    }
}
