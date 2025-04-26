package com.ib.client;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

class EWrapperMsgGenerator_tickEFP_5_0_Test {

    @ParameterizedTest
    @CsvSource({ "1, 1, 1.2, \"1.2\", 1.3, 2, \"2024-10-27\", 1.4, 1.5", "100, 2, 2.5, \"2.5\", 2.6, 5, \"2024-12-15\", 2.7, 2.8", "5, 3, 3.7, \"3.7\", 3.8, 1, \"2023-03-10\", 3.9, 4.0", "10, 4, 0.1, \"0.1\", 0.2, 10, \"2024-06-01\", 0.3, 0.4" })
    void testTickEFP(int tickerId, int tickType, double basisPoints, String formattedBasisPoints, double impliedFuture, int holdDays, String futureExpiry, double dividendImpact, double dividendsToExpiry) {
        String expected = "id=" + tickerId + "  " + TickType.getField(tickType) + ": basisPoints = " + basisPoints + "/" + formattedBasisPoints + " impliedFuture = " + impliedFuture + " holdDays = " + holdDays + " futureExpiry = " + futureExpiry + " dividendImpact = " + dividendImpact + " dividends to expiry = " + dividendsToExpiry;
        String actual = EWrapperMsgGenerator.tickEFP(tickerId, tickType, basisPoints, formattedBasisPoints, impliedFuture, holdDays, futureExpiry, dividendImpact, dividendsToExpiry);
        assertEquals(expected, actual);
    }

    // Helper class for mocking TickType (crucial for complete branch coverage)
    static class TickType {

        static String getField(int tickType) {
            switch(tickType) {
                case 1:
                    return "BID";
                case 2:
                    return "ASK";
                case 3:
                    return "LAST";
                case 4:
                    return "HIGH";
                // Handles unexpected tickType values
                default:
                    return "UNKNOWN";
            }
        }
    }
}
