package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class AnyWrapperMsgGenerator_error_0_0_Test {

    // Test class
    @Test
    public void test_error_Exception() {
        // Create an instance of the class
        AnyWrapperMsgGenerator anyWrapperMsgGenerator = new AnyWrapperMsgGenerator();
        // Invoke the method under test
        String result = anyWrapperMsgGenerator.error(new Exception("Exception"));
        // Assert the result
        assertEquals("Error - Exception", result);
    }
}
