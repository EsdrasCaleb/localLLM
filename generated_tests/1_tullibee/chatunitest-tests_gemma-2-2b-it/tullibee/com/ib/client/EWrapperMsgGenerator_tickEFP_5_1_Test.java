package com.ib.client;

import java.util.HashMap;
import java.util.Map;
import static org.mockito.ArgumentMatchers.anyInt;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.text.DateFormat;
import java.util.Date;
import java.util.Vector;

public class EWrapperMsgGenerator_tickEFP_5_1_Test {

    @Test
    void testTickEFP() {
        EWrapperMsgGenerator eWrapperMsgGenerator = mock(EWrapperMsgGenerator.class);
        // Arrange
        int tickerId = 123;
        int tickType = 1;
        double basisPoints = 0.01;
        String formattedBasisPoints = "1.00";
        double impliedFuture = 0.01;
        int holdDays = 0;
        String futureExpiry = "2023-01-01";
        double dividendImpact = 0.01;
        double dividendsToExpiry = 0.01;
        // Act
        String result = eWrapperMsgGenerator.tickEFP(tickerId, tickType, basisPoints, formattedBasisPoints, impliedFuture, holdDays, futureExpiry, dividendImpact, dividendsToExpiry);
        // Assert
        assertEquals("id=123  TickType.getField(1): basisPoints = 1.00/1.00 impliedFuture = 0.01 holdDays = 0 futureExpiry = 2023-01-01 dividendImpact = 0.01 dividends to expiry = 0.01", result);
    }
}
