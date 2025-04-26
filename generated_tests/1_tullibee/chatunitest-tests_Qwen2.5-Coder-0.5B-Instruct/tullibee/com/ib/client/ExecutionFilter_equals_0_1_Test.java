package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ExecutionFilter_equals_0_1_Test {

    @Test
    public void testEquals() {
        // Create mock instance of ExecutionFilter
        ExecutionFilter mockFilter = Mockito.mock(ExecutionFilter.class);
        // Set expected values for client ID, acctCode, time, symbol, secType, exchange, and side
        mockFilter.m_clientId = 12345;
        mockFilter.m_acctCode = "ABC123";
        mockFilter.m_time = "2023-04-01T12:00:00Z";
        mockFilter.m_symbol = "AAPL";
        mockFilter.m_secType = "BID";
        mockFilter.m_exchange = "NASDAQ";
        mockFilter.m_side = "BUY";
        // Call the equals method on mockFilter with another instance
        boolean result = mockFilter.equals(mockFilter);
        // Assert that the result is true
        assertEquals(true, result);
    }
}
