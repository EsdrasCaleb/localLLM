package net.kencochrane.a4j;

import net.kencochrane.a4j.DAO.Search;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.beans.*;

class A4j_DirectorSearch_6_3_Test {

    @Test
    void DirectorSearchTest() {
        // given
        String directorName = "John Doe";
        String mode = "testMode";
        String page = "1";
        // create mock objects
        Search search = Mockito.mock(Search.class);
        // when
        ProductInfo productInfo = new ProductInfo();
        Mockito.when(search.DirectorSearch(directorName, mode, page)).thenReturn(productInfo);
        // create A4j object
        A4j a4j = new A4j();
        // setSearch(search);
        // then
        ProductInfo result = a4j.DirectorSearch(directorName, mode, page);
        assertSame(productInfo, result);
    }
}
