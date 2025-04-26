package com.ib.client;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class ScannerSubscription_maturityDateAbove_14_0_Test {

    @InjectMocks
    private ScannerSubscription scannerSubscription;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testMaturityDateAbove() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        String expectedMaturityDate = "2023-12-31";
        setPrivateField(scannerSubscription, "m_maturityDateAbove", expectedMaturityDate);
        // Act
        String actualMaturityDate = scannerSubscription.maturityDateAbove();
        // Assert
        assertEquals(expectedMaturityDate, actualMaturityDate);
    }

    @Test
    public void testMaturityDateAbove_WhenNull() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        setPrivateField(scannerSubscription, "m_maturityDateAbove", null);
        // Act
        String actualMaturityDate = scannerSubscription.maturityDateAbove();
        // Assert
        assertNull(actualMaturityDate);
    }

    private void setPrivateField(Object obj, String fieldName, Object value) throws NoSuchFieldException, IllegalAccessException {
        Field field = obj.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(obj, value);
    }
}
