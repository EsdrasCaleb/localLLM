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

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_tickString_4_0_Test {

    // Test class
    @Test
    public void testTickString() {
        String expected = "id=1  BID=100.0";
        String actual = EWrapperMsgGenerator.tickString(1, TickType.BID, "100.0");
        assertEquals(expected, actual);
    }
}
