package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class ComboLeg_equals_0_0_Test {

    @ExtendWith(MockitoExtension.class)
    public class TestComboLegEquals0_0 {

        @Mock
        private ComboLeg comboLeg;

        @Test
        public void testEqualsObject() {
            // Create another instance of ComboLeg
            ComboLeg otherComboLeg = new ComboLeg(1, 1, "buy", "exchange", 2, 3, "location");
            // <Buggy Line>: cannot find symbol  symbol:   variable comboLeg  location: class com.ib.client.ComboLeg_equals_0_0_Test
            assertEquals(comboLeg, otherComboLeg);
        }
    }
}
