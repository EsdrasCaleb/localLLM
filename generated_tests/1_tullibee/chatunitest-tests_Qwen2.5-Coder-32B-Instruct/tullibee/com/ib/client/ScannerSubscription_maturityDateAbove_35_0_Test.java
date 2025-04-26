package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateAbove_35_0_Test {

    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        scannerSubscription = new ScannerSubscription();
    }

    @Test
    public void testMaturityDateAbove() throws NoSuchFieldException, IllegalAccessException {
        // Given
        String testDate = "20231231";
        assertNull(getMaturityDateAbove(scannerSubscription), "Initial value should be null");
        // When
        scannerSubscription.maturityDateAbove(testDate);
        // Then
        assertEquals(testDate, getMaturityDateAbove(scannerSubscription), "m_maturityDateAbove should be set to the provided date");
    }

    private String getMaturityDateAbove(ScannerSubscription subscription) throws NoSuchFieldException, IllegalAccessException {
        Field field = ScannerSubscription.class.getDeclaredField("m_maturityDateAbove");
        field.setAccessible(true);
        return (String) field.get(subscription);
    }
}
