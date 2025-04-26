package net.kencochrane.a4j;

// Test class
import org.junit.Test;
import static org.junit.Assert.assertEquals;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

public class A4j_BlendedSearch_1_0_Test {

    @Test
    public void testBlendedSearch() {
        A4j a4j = new A4j();
        BlendedSearch actual = a4j.BlendedSearch("test", "test");
        assertEquals("BlendedSearch", actual.getClass().getSimpleName());
    }
}
