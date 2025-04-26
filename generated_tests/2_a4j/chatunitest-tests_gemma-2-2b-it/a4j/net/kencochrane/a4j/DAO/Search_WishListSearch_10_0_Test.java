package net.kencochrane.a4j.DAO;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.wutka.jox.JOXBeanInputStream;
import net.kencochrane.a4j.beans.BlendedSearch;
import net.kencochrane.a4j.beans.ProductInfo;
import net.kencochrane.a4j.beans.SellerSearch;
import net.kencochrane.a4j.file.FileUtil;
import java.io.FileInputStream;

public class Search_WishListSearch_10_0_Test {

    @Test
    void WishListSearchTest() {
        Search search = Mockito.spy(new Search());
        Mockito.when(search.Generic("WishlistSearch", "123", "mode", "lite", "1", "all")).thenReturn(new ProductInfo());
        ProductInfo result = search.WishListSearch("123");
        assertEquals(result, new ProductInfo());
    }
}
