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

public class EWrapperMsgGenerator_tickString_4_0_Test {

    @Test
    public void testTickString() {
        int tickerId = 123;
        int tickType = 456;
        String value = "testValue";
        String expectedResult = "id=123  TickType.getField(456)=testValue";
        assertEquals(expectedResult, EWrapperMsgGenerator.tickString(tickerId, tickType, value));
    }
}
