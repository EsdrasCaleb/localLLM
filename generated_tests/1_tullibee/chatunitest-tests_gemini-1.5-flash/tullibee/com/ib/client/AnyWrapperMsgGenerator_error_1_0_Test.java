package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class AnyWrapperMsgGenerator_error_1_0_Test {

    @Test
    void testErrorNullString() {
        assertNull(AnyWrapperMsgGenerator.error((String) null));
    }

    @Test
    void testErrorEmptyString() {
        assertEquals("", AnyWrapperMsgGenerator.error(""));
    }

    @Test
    void testErrorNotEmptyString() {
        String testString = "This is a test string";
        assertEquals(testString, AnyWrapperMsgGenerator.error(testString));
    }

    @Test
    void testErrorWhitespaceString() {
        assertEquals("   ", AnyWrapperMsgGenerator.error("   "));
    }

    @Test
    void testErrorException() {
        Exception e = new Exception("Test Exception");
        String expectedMessage = "Test Exception";
        assertEquals(expectedMessage, AnyWrapperMsgGenerator.error(e));
    }

    @Test
    void testErrorNullException() {
        assertNull(AnyWrapperMsgGenerator.error((Exception) null));
    }
}
