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

public class Search_WishListSearch_10_2_Test {

    @Mock
    Search search;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testWishListSearch() {
        // Arrange
        String wishListId = "wishlistId";
        ProductInfo expectedProductInfo = new ProductInfo();
        Mockito.when(search.WishListSearch(wishListId)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo actualProductInfo = search.WishListSearch(wishListId);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
    }
}
