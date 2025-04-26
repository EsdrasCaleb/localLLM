package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class AnyWrapperMsgGenerator_connectionClosed_3_0_Test {

    @Test
    void testConnectionClosed() {
        String expected = "Connection Closed";
        String actual = AnyWrapperMsgGenerator.connectionClosed();
        assertEquals(expected, actual);
    }
}
