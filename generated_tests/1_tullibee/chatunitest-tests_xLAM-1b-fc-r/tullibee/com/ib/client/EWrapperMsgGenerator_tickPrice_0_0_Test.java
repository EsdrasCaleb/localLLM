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

public class EWrapperMsgGenerator_tickPrice_0_0_Test {

    @Test
    public void testTickPrice() {
        EWrapperMsgGenerator eWrapperMsgGenerator = new EWrapperMsgGenerator();
        int tickerId = 123;
        int field = 456;
        double price = 789.0;
        int canAutoExecute = 1;
        String expectedResult = "id=" + tickerId + "  " + TickType.getField(field) + "=" + price + " " + ((canAutoExecute != 0) ? "canAutoExecute" : "noAutoExecute");
        assertEquals(expectedResult, eWrapperMsgGenerator.tickPrice(tickerId, field, price, canAutoExecute));
    }
}
