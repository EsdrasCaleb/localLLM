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
public class EWrapperMsgGenerator_tickOptionComputation_2_0_Test {

    @Mock
    private EWrapperMsgGenerator focal;

    @InjectMocks
    private EWrapperMsgGenerator testObject;

    @Test
    public void testTickOptionComputation() {
        // Given
        int tickerId = 1;
        int field = 0;
        double impliedVol = 100.0;
        double delta = 0.1;
        double modelPrice = 100.0;
        double pvDividend = 10.0;
        // When
        String result = focal.tickOptionComputation(tickerId, field, impliedVol, delta, modelPrice, pvDividend);
        // Then
        assertEquals("id=1  vol=100.0 delta=0.1 modelPrice=100.0 pvDividend=10.0", result);
    }
}
