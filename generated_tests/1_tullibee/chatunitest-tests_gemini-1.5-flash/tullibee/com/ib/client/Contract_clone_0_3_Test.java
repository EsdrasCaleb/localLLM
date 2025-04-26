package com.ib.client;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Vector;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Contract_clone_0_3_Test {

    @Test
    void testClone() throws CloneNotSupportedException, NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        // Create a sample Contract object
        Contract originalContract = new Contract();
        originalContract.m_conId = 123;
        originalContract.m_symbol = "AAPL";
        originalContract.m_comboLegs.add("leg1");
        originalContract.m_comboLegs.add("leg2");
        // Invoke the clone method using reflection (because it's public)
        Method cloneMethod = Contract.class.getMethod("clone");
        Contract clonedContract = (Contract) cloneMethod.invoke(originalContract);
        // Assertions to check if the clone is a deep copy
        Assertions.assertNotSame(originalContract, clonedContract, "Clone should be a different object");
        Assertions.assertEquals(originalContract.m_conId, clonedContract.m_conId, "conId should be the same");
        Assertions.assertEquals(originalContract.m_symbol, clonedContract.m_symbol, "symbol should be the same");
        Assertions.assertNotSame(originalContract.m_comboLegs, clonedContract.m_comboLegs, "comboLegs should be a different object");
        Assertions.assertEquals(originalContract.m_comboLegs.size(), clonedContract.m_comboLegs.size(), "comboLegs size should be the same");
        Assertions.assertEquals(originalContract.m_comboLegs.get(0), clonedContract.m_comboLegs.get(0), "comboLegs content should be the same");
        Assertions.assertEquals(originalContract.m_comboLegs.get(1), clonedContract.m_comboLegs.get(1), "comboLegs content should be the same");
        // Test with empty comboLegs
        Contract originalContract2 = new Contract();
        originalContract2.m_conId = 456;
        originalContract2.m_symbol = "MSFT";
        Contract clonedContract2 = (Contract) cloneMethod.invoke(originalContract2);
        Assertions.assertNotSame(originalContract2, clonedContract2);
        Assertions.assertEquals(originalContract2.m_conId, clonedContract2.m_conId);
        Assertions.assertEquals(originalContract2.m_symbol, clonedContract2.m_symbol);
        Assertions.assertEquals(originalContract2.m_comboLegs.size(), clonedContract2.m_comboLegs.size());
    }
}
