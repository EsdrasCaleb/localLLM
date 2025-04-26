package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class ScannerSubscription_moodyRatingBelow_11_0_Test {

    // Rest of the code
    // Focal method
    private String m_moodyRatingBelow;

    public String moodyRatingBelow() {
        // <Buggy Line>: cannot find symbol  symbol:   variable m_moodyRatingBelow  location: class com.ib.client.ScannerSubscription_moodyRatingBelow_11_0_Test
        return m_moodyRatingBelow;
    }

    @Test
    public void testMoodyRatingBelow() {
        ScannerSubscription_moodyRatingBelow_11_0_Test testClass = Mockito.spy(new ScannerSubscription_moodyRatingBelow_11_0_Test());
        String expected = "AA";
        String actual = testClass.moodyRatingBelow();
        assertEquals(expected, actual);
    }
}
