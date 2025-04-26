package com.ib.client;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Contract_clone_0_2_Test {

    @Mock
    private Vector m_comboLegs;

    @Test
    public void testClone() throws CloneNotSupportedException {
        // Arrange
        Contract contract = new Contract(1, "AAPL", "STK", "2022", 100, "C", "C", "NYMEX", "USD", "AAPL", m_comboLegs, "BATS", true, "US", "123456");
        // <Buggy Line>: incompatible types: java.lang.Object cannot be converted to com.ib.client.Contract
        Contract clonedContract = (Contract) contract.clone();
        // Assert
        when(contract.m_comboLegs).thenReturn(m_comboLegs);
        when(m_comboLegs.clone()).thenReturn(m_comboLegs);
        assertSame(contract, clonedContract);
        assertSame(m_comboLegs, clonedContract.m_comboLegs);
    }

    @Test
    public void testCloneThrowsCloneNotSupportedException() {
        // Arrange
        Contract contract = new Contract(1, "AAPL", "STK", "2022", 100, "C", "C", "NYMEX", "USD", "AAPL", new Vector(), "BATS", true, "US", "123456");
        // Act and Assert
        assertThrows(CloneNotSupportedException.class, () -> contract.clone());
    }
}
