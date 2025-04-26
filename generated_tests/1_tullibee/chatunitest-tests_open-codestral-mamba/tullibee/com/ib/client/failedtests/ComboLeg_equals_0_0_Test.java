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

    @Test
    public void testEquals_DifferentClass_ReturnsFalse() {
        assertFalse(comboLeg.equals(new Object()));
    }

    @Test
    public void testEquals_SameAttributes_ReturnsTrue() {
        otherComboLeg.m_conId = comboLeg.m_conId;
        otherComboLeg.m_ratio = comboLeg.m_ratio;
        otherComboLeg.m_openClose = comboLeg.m_openClose;
        otherComboLeg.m_shortSaleSlot = comboLeg.m_shortSaleSlot;
        otherComboLeg.m_action = comboLeg.m_action;
        otherComboLeg.m_exchange = comboLeg.m_exchange;
        otherComboLeg.m_designatedLocation = comboLeg.m_designatedLocation;
        when(utilMock.StringCompareIgnCase(anyString(), anyString())).thenReturn(0);
        assertTrue(comboLeg.equals(otherComboLeg));
    }

    @Test
    public void testEquals_DifferentAttributes_ReturnsFalse() {
        otherComboLeg.m_conId = comboLeg.m_conId + 1;
        when(utilMock.StringCompareIgnCase(anyString(), anyString())).thenReturn(0);
        assertFalse(comboLeg.equals(otherComboLeg));
    }
}
