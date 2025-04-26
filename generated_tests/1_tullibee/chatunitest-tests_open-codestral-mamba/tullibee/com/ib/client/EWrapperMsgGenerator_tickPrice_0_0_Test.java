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

class EWrapperMsgGenerator_tickPrice_0_0_Test {

    @Test
    void testTickPrice() {
        int tickerId = 1;
        int field = 2;
        double price = 100.50;
        int canAutoExecute = 1;
        String expected = "id=" + tickerId + "  " + TickType.getField(field) + "=" + price + " canAutoExecute";
        String actual = EWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute);
        assertEquals(expected, actual);
    }
}
