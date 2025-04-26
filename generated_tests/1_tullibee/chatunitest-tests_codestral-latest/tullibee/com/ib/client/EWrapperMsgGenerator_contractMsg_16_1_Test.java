package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.text.DateFormat;
import java.util.Date;

@ExtendWith(MockitoExtension.class)
class EWrapperMsgGenerator_contractMsg_16_1_Test {

    @Test
    void testClone() throws CloneNotSupportedException {
        Vector<ComboLeg> comboLegs = new Vector<>();
        comboLegs.add(new ComboLeg());
        Contract original = new Contract(1, "AAPL", "STK", "20231231", 150.0, "C", "100", "NYSE", "USD", "AAPL", comboLegs, "ARCA", true, "ISIN", "US0378331005");
        Contract cloned = (Contract) original.clone();
        assertNotSame(original, cloned);
        assertEquals(original, cloned);
        assertNotSame(original.m_comboLegs, cloned.m_comboLegs);
    }

    @Test
    void testEquals() {
        Vector<ComboLeg> comboLegs1 = new Vector<>();
        comboLegs1.add(new ComboLeg());
        Vector<ComboLeg> comboLegs2 = new Vector<>();
        comboLegs2.add(new ComboLeg());
        Contract contract1 = new Contract(1, "AAPL", "STK", "20231231", 150.0, "C", "100", "NYSE", "USD", "AAPL", comboLegs1, "ARCA", true, "ISIN", "US0378331005");
        Contract contract2 = new Contract(1, "AAPL", "STK", "20231231", 150.0, "C", "100", "NYSE", "USD", "AAPL", comboLegs2, "ARCA", true, "ISIN", "US0378331005");
        assertEquals(contract1, contract2);
        Contract contract3 = new Contract(2, "GOOGL", "STK", "20231231", 150.0, "C", "100", "NYSE", "USD", "GOOGL", comboLegs1, "ARCA", true, "ISIN", "US0378331005");
        assertNotEquals(contract1, contract3);
    }
}
