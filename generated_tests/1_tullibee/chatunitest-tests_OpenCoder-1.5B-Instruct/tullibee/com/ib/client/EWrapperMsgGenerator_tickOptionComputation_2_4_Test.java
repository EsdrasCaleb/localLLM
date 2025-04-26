// Test method
package com.ib.client;

import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class EWrapperMsgGenerator_tickOptionComputation_2_4_Test {

    @Mock
    private EWrapper eWrapper;

    @Test
    public void testTickOptionComputation() {
        int tickerId = 1;
        int field = TickType.CURRENT_LAST.ordinal();
        double impliedVol = 0.5;
        double delta = 0.1;
        double modelPrice = 100.0;
        double pvDividend = 10.0;
        String expectedMessage = "id=" + tickerId + "  " + TickType.CURRENT_LAST + ": vol = " + modelPrice + ": vol = " + pvDividend;
        String resultMessage = tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        assertEquals(expectedMessage, resultMessage);
    }

    public String tickOptionComputation(int tickerId, int field, double impliedVol, double delta, double modelPrice, double pvDividend) {
        return "id=" + tickerId + "  " + TickType.getTickType(field) + ": vol = " + modelPrice + ": vol = " + pvDividend;
    }

    public enum TickType {

        CURRENT_LAST;

        public static String getTickType(int field) {
            return TickType.values()[field].name();
        }
    }
}
