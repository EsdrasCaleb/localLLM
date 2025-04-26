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

public class EWrapperMsgGenerator_tickSize_1_3_Test {

    @Test
    public void testTickSize() {
        int tickerId = 1;
        int field = 2;
        int size = 3;
        String expected = "id=1  field=2 value=3";
        String actual = EWrapperMsgGenerator.tickSize(tickerId, field, size);
        assertEquals(expected, actual);
    }
}
