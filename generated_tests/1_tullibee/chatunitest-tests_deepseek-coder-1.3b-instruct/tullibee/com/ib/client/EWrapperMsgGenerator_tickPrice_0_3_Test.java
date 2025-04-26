package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_tickPrice_0_3_Test {

    @InjectMocks
    private EWrapperMsgGenerator eWrapperMsgGenerator;

    @Test
    public void testTickPrice() {
        int tickerId = 1;
        int field = 2;
        double price = 3.0;
        int canAutoExecute = 1;
        String expected = "id=" + tickerId + "  " + TickType.getField(field) + "=" + price + " " + ((canAutoExecute != 0) ? " canAutoExecute" : " noAutoExecute");
        String result = eWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute);
        assertEquals(expected, result);
    }
}
