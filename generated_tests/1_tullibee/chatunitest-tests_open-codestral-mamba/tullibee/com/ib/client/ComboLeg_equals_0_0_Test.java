package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ComboLeg_equals_0_0_Test {

    @Mock
    private Util utilMock;

    @InjectMocks
    private ComboLeg comboLeg;

    private ComboLeg otherComboLeg;

    @BeforeEach
    public void setUp() {
        otherComboLeg = new ComboLeg();
    }

    @Test
    public void testEquals_SameObject_ReturnsTrue() {
        assertTrue(comboLeg.equals(comboLeg));
    }

    @Test
    public void testEquals_Null_ReturnsFalse() {
        assertFalse(comboLeg.equals(null));
    }
}
