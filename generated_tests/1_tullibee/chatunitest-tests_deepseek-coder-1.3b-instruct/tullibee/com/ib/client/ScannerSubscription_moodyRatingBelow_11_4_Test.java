package com.ib.client;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_moodyRatingBelow_11_4_Test {

    private ScannerSubscription m_scannerSubscription;

    @BeforeEach
    public void setUp() {
        m_scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMoodyRatingBelow() {
        String expected = "Expected Result";
        m_scannerSubscription.moodyRatingBelow("Expected Result");
        assertEquals(expected, m_scannerSubscription.moodyRatingBelow());
    }
}
