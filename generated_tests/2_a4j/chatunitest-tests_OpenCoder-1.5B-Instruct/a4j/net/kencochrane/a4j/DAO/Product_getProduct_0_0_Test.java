package net.kencochrane.a4j.DAO;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.wutka.jox.JOXBeanInputStream;
import net.kencochrane.a4j.beans.*;
import net.kencochrane.a4j.file.FileUtil;

class Product_getProduct_0_0_Test {

    @Test
    void getProductTest() throws FileNotFoundException {
        // Arrange
        String asin = "123456789012";
        String offer = "bestseller";
        String page = "1";
        Product product = new Product();
        FileUtil fileUtil = Mockito.mock(FileUtil.class);
        JOXBeanInputStream joxIn = Mockito.mock(JOXBeanInputStream.class);
        ProductInfo productInfoBean = Mockito.mock(ProductInfo.class);
        ProductDetails productDetails = Mockito.mock(ProductDetails.class);
        ProductDetails accessoryProductDetails = Mockito.mock(ProductDetails.class);
        Accessories accessories = Mockito.mock(Accessories.class);
        ArrayList accessoryArray = new ArrayList();
        ArrayList detailsArray = new ArrayList();
        MiniProduct miniProduct = Mockito.mock(MiniProduct.class);
        Search search = Mockito.mock(Search.class);
        FileInputStream fileIn = Mockito.mock(FileInputStream.class);
        List<ProductDetails> detailsList = new ArrayList<>();
        List<MiniProduct> accessoryList = new ArrayList<>();
        // Act
        product.getProduct(asin, offer, page);
        // Assert
        assertNotNull(product);
    }
}
