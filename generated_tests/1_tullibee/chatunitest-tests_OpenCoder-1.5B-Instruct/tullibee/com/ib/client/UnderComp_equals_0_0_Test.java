package com.ib.client;

import java.util.Objects;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class UnderComp_equals_0_0_Test {

    public static class UnderComp {

        public int m_conId;

        public double m_delta;

        public double m_price;

        public UnderComp() {
        }

        public UnderComp(int conId, double delta, double price) {
            this.m_conId = conId;
            this.m_delta = delta;
            this.m_price = price;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (obj == null || !(obj instanceof UnderComp)) {
                return false;
            }
            UnderComp l_theOther = (UnderComp) obj;
            return m_conId == l_theOther.m_conId && m_delta == l_theOther.m_delta && m_price == l_theOther.m_price;
        }

        @Override
        public int hashCode() {
            return Objects.hash(m_conId, m_delta, m_price);
        }
    }

    @Test
    public void testEquals() {
        UnderComp obj1 = new UnderComp(123, 0.5, 100.0);
        UnderComp obj2 = new UnderComp(123, 0.5, 100.0);
        UnderComp obj3 = new UnderComp(456, 0.5, 100.0);
        assertTrue(obj1.equals(obj2));
        assertFalse(obj1.equals(obj3));
    }
}
