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

public class A4j_AuthorSearch_5_3_Test {

    @Test
    void AuthorSearch() {
        A4j a4j = new A4j();
        ProductInfo productInfo = a4j.AuthorSearch("authorName", "page");
        assertEquals(productInfo, new ProductInfo());
    }
}
