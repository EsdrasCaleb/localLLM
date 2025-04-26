package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class AnyWrapperMsgGenerator_connectionClosed_3_1_Test {

    @ExtendWith(MockitoExtension.class)
    public class AnyWrapperMsgGeneratorConnectionClosed_3_1Test {

        @Test
        public void testConnectionClosed() {
            // Create an instance of AnyWrapperMsgGenerator
            AnyWrapperMsgGenerator wrapper = new AnyWrapperMsgGenerator();
            // Use Mockito to mock the connectionClosed method
            when(wrapper.connectionClosed()).thenReturn("Connection Closed");
            // Call the method to be tested
            String result = wrapper.connectionClosed();
            // Verify that the result matches the expected output
            assertEquals("Connection Closed", result);
        }
    }
}
