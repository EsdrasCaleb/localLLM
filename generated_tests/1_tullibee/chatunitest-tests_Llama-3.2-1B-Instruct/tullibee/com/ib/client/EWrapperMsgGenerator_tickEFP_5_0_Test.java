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
public class EWrapperMsgGenerator_tickEFP_5_0_Test {

    @Mock
    private EWrapperMsgGenerator focal;

    @InjectMocks
    private EWrapperMsgGenerator instance;

    @Test
    public void testTickEFP() {
        // Given
        int tickerId = 123;
        int tickType = 1;
        double basisPoints = 10.0;
        String formattedBasisPoints = "10.00";
        double impliedFuture = 100.0;
        int holdDays = 30;
        String futureExpiry = "2022-01-01";
        double dividendImpact = 0.05;
        double dividendsToExpiry = 10.0;
        // When
        String result = focal.tickEFP(tickerId, tickType, basisPoints, formattedBasisPoints, impliedFuture, holdDays, futureExpiry, dividendImpact, dividendsToExpiry);
        // Then
        assertEquals("id=123  field=1: basisPoints = 10.00/10.00 impliedFuture = 100.0 holdDays = 30 futureExpiry = 2022-01-01 dividendImpact = 0.05 dividends to expiry = 10.0", result);
    }
}
