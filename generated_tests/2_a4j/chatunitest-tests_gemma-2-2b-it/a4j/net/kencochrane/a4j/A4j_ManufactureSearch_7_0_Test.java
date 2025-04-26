package net.kencochrane.a4j;

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

public class A4j_ManufactureSearch_7_0_Test {

    @Test
    void ManufactureSearch() {
        A4j a4j = new A4j();
        ProductInfo result = a4j.ManufactureSearch("ABC", "Mode", "Page");
        assertEquals(result, new ProductInfo());
    }
}
