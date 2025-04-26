package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Vector;

@ExtendWith(MockitoExtension.class)
public class Contract_equals_1_1_Test {

    @Mock
    private UnderComp underComp;

    @InjectMocks
    private Contract contract;

    @Test
    public void testEquals_SameInstance() {
        assertTrue(contract.equals(contract));
    }

    @Test
    public void testEquals_DifferentClass() {
        assertFalse(contract.equals("not a Contract"));
    }

    @Test
    public void testEquals_DifferentContract() {
        Contract otherContract = new Contract(1, "ABC", "BOND", "2024-03-20", 100, "CASH", "EURO", "LONDON", "EUR", "ABC", new Vector(), "SPX", false, "STK", "123456789");
        assertFalse(contract.equals(otherContract));
    }

    @Test
    public void testEquals_DifferentStrike() {
        Contract otherContract = new Contract(1, "ABC", "BOND", "2024-03-20", 101, "CASH", "EURO", "LONDON", "EUR", "ABC", new Vector(), "SPX", false, "STK", "123456789");
        assertFalse(contract.equals(otherContract));
    }
}
