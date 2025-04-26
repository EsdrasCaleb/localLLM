package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

class EWrapperMsgGenerator_tickSize_1_0_Test {

    @Test
    void tickSize() {
        int tickerId = 123;
        int field = 456;
        int size = 789;
        String expected = "id=123  FIELD=456=789";
        String result = EWrapperMsgGenerator.tickSize(tickerId, field, size);
        assertEquals(expected, result);
    }
}
