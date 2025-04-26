package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class TickType_getField_0_0_Test {

    @Test
    void testGetField_ValidTickType() {
        assertEquals("bidSize", TickType.getField(TickType.BID_SIZE));
    }
}
