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
public class EWrapperMsgGenerator_tickPrice_0_0_Test {

    @Test
    public void testTickPrice() {
        int tickerId = 123456;
        int field = 0;
        double price = 1.23456;
        int canAutoExecute = 0;
        String result = EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute);
        assertEquals("id=123456  Bid=1.23456 noAutoExecute", result);
    }
}
